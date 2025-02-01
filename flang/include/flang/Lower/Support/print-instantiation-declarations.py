# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This script generates declarations of hash calculations and equality checks
# from flang/include/flang/Lower/Support/Utils.h
# The total number of implicitly instantiated functions was in excess of 7000
# consuming large amounts of memory at compile time.
#
# The script generates declarations of functions which take operands in the
# form of
#    Operation<Type<TypeCategory, Kind>>     (*)
#
# The main utility is an category-kind iterator, which generates valid pairs
# of TypeCategory and Kind.
# This is then used to generate strings with operations in the form of (*).
# These strings, in turn, are then used to generate the function declarations.

MacroName = "INSTANTIATION"


class Category:
    def __init__(self, name, kinds):
        self.name = name
        self.kinds = kinds
        assert isinstance(kinds, list) and kinds, "kinds should be a nonempty list"

    def to_string(self):
        return f"Fortran::common::TypeCategory::{self.name}"

    def __repr__(self):
        return self.name


Character = Category("Character", [1, 2, 4])
Complex = Category("Complex", [2, 3, 4, 8, 10, 16])
Integer = Category("Integer", [1, 2, 4, 8, 16])
Logical = Category("Logical", [1, 2, 4, 8])
Real = Category("Real", [2, 3, 4, 8, 10, 16])
Unsigned = Category("Unsigned", [1, 2, 4, 8, 16])

Categories = [Character, Complex, Integer, Logical, Real, Unsigned]


# The main implementation of the category-kind iterator. Implements __next__.
class CKIteratorImpl:
    def __init__(self, *, categories=..., kinds=...):
        self.categories = categories
        self.kinds = kinds
        self.category_index = 0
        self.kind_index = -1
        self.at_end = False

    def _dereference(self):
        category = Categories[self.category_index]
        kind = category.kinds[self.kind_index]
        return category, kind

    def _is_allowed(self):
        assert not self.at_end
        category, kind = self._dereference()
        if self.kinds is not ... and kind not in self.kinds:
            return False
        if self.categories is not ... and category not in self.categories:
            return False
        return True

    def _increment(self):
        if self.at_end:
            return

        self.kind_index += 1
        category = Categories[self.category_index]

        if self.kind_index >= len(category.kinds):
            self.kind_index = 0
            self.category_index += 1

        if self.category_index >= len(Categories):
            self.at_end = True

    def __next__(self):
        if self.at_end:
            raise StopIteration()

        while not self.at_end:
            self._increment()
            if self.at_end:
                raise StopIteration()
            if self._is_allowed():
                break

        return self._dereference()


# User interface of the category-kind iterator. Implements __iter__.
class CKIterator:
    def __init__(self, *, categories=..., kinds=...):
        self.categories = categories  # Allowed categories
        self.kinds = kinds  # Allowed kinds

    def __iter__(self):
        return CKIteratorImpl(categories=self.categories, kinds=self.kinds)


class Type:
    def __init__(self, category, kind):
        self.category = category
        self.kind = kind

    def to_string(self):
        return f"Fortran::evaluate::Type<{self.category.to_string()}, {self.kind}>"


ArithCategories = [Complex, Integer, Real, Unsigned]


class Op:
    def __init__(self, name, categories=..., kinds=...):
        self.name = name
        self.categories = categories
        self.kinds = kinds

    def to_strings(self, category, kind):
        return [f"Fortran::evaluate::{self.name}<{Type(category, kind).to_string()}>"]

    def __repr__(self):
        return self.name


class ConvertOp(Op):
    # Conversion classes: for each category c in class: Convert(Type(c), c)
    Classes = [
        [Character],
        [Complex],
        [Integer, Real, Unsigned],
        [Logical],
    ]

    def __init__(self, categories=..., kinds=...):
        super().__init__("Convert", categories, kinds)

    def to_strings(self, category, kind):
        for cls in ConvertOp.Classes:
            if not category in cls:
                continue
            cvt_strings = [
                f"Fortran::evaluate::{self.name}"
                f"<{Type(category, kind).to_string()}, {to_c.to_string()}>"
                for to_c in cls
            ]
            return cvt_strings


GeneralOps = [
    Op(name)
    for name in [
        "ArrayConstructor",
        "Constant",
        "Designator",
        "Expr",
        "FunctionRef",
        "Parentheses",
    ]
]

MathOps = [
    Op("Add", ArithCategories),
    Op("Divide", ArithCategories),
    Op("Multiply", ArithCategories),
    Op("Negate", ArithCategories),
    Op("Subtract", ArithCategories),
    Op("Extremum", [Character, Integer, Real, Unsigned]),
    Op("Power", [Complex, Integer, Real]),
    Op("RealToIntPower", [Complex, Real]),
]

OtherOps = [
    Op("Relational", [Character, Complex, Integer, Real, Unsigned]),
    ConvertOp(),
]


# Construct a dictionary:
#   (category, kind) -> {
#     {op: [string-for-op-with-category-kind]}
#   }
# for all valid combinations of category and kind.
#
# The entry for a given pair (category, kind) is another dictionary that
# is indexed by operations, and contains the strings representing that
# operation with the given category and kind. Usually there is only one
# string like that, e.g.
#   {Add: ["Add<TypeCategory::Integer, 8>"]}
# with the exception of the Convert operation, which contains extra category
# parameter (but without an extra kind), e.g.:
#   Convert<Type<TypeCategory::Integer, 4>, TypeCategory::Integer>
#   Convert<Type<TypeCategory::Integer, 4>, TypeCategory::Real>
#   Convert<Type<TypeCategory::Integer, 4>, TypeCategory::Unsigned>
# In such case the list of strings will contain all of these.


def build_dict():
    opstr_dict = {}

    def update(c, k, op, strings):
        if entry := opstr_dict.get((c, k)):
            if ss := entry.get(op):
                ss.append(strings)
            else:
                entry[op] = strings
        else:
            opstr_dict[c, k] = {op: strings}

    for op in [*GeneralOps, *MathOps, *OtherOps]:
        for c, k in CKIterator(categories=op.categories):
            update(c, k, op, op.to_strings(c, k))

    return opstr_dict


def print_getHashValue(opstr_dict):
    format_string = (
        f"{MacroName}(unsigned int Fortran::lower::HashEvaluateExpr::"
        f"getHashValue(const {{0}} &));"
    )

    for (c, k), entry in opstr_dict.items():
        for op, strings in entry.items():
            if op.name in ["FunctionRef"]:
                continue
            for s in strings:
                print(format_string.format(s))


def print_isBinaryEqual(opstr_dict):
    format_string = (
        f"{MacroName}(bool Fortran::lower::IsEqualEvaluateExpr::"
        f"isBinaryEqual(const {{0}} &, const {{0}} &));"
    )

    for (c, k), entry in opstr_dict.items():
        for op, strings in entry.items():
            if op not in MathOps or op.name in ["Negate"]:
                continue
            for s in strings:
                print(format_string.format(s))


def print_isEqual(opstr_dict):
    format_string = (
        f"{MacroName}(bool Fortran::lower::IsEqualEvaluateExpr::"
        f"isEqual(const {{0}} &, const {{1}} &));"
    )

    for (c0, k), entry in opstr_dict.items():
        for op0, strings0 in entry.items():
            for op1, strings1 in entry.items():
                # Expr and Relational are handled separately
                if op0.name in ["Expr", "Relational"]:
                    continue
                if op1.name in ["Expr", "Relational"]:
                    continue
                # FunctionRef does not appear with another FunctionRef.
                if op0.name == "FunctionRef" and op1.name == "FunctionRef":
                    continue
                for s0, s1 in [(x, y) for x in strings0 for y in strings1]:
                    print(format_string.format(s0, s1))

    # Expr and Relational are only paired with themselves, but across
    # different sets of types:
    # - Expr is over kinds within the same category,
    # - Relational is over all types
    for (c0, k0), entry0 in opstr_dict.items():
        for op0, strings0 in entry0.items():
            if op0.name not in ["Expr", "Relational"]:
                continue
            for (c1, k1), entry1 in opstr_dict.items():
                for op1, strings1 in entry1.items():
                    if op0.name != op1.name:
                        continue
                    if op0.name == "Expr" and c0 != c1:
                        continue
                    for s0, s1 in [(x, y) for x in strings0 for y in strings1]:
                        print(format_string.format(s0, s1))


opstr_dict = build_dict()

print_getHashValue(opstr_dict)
print_isBinaryEqual(opstr_dict)
print_isEqual(opstr_dict)
