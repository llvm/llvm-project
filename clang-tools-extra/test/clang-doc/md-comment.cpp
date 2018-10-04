// THIS IS A GENERATED TEST. DO NOT EDIT.
// To regenerate, see clang-doc/gen_test.py docstring.
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"

/// \brief Brief description.
///
/// Extended description that
/// continues onto the next line.
/// 
/// <ul class="test">
///   <li> Testing.
/// </ul>
///
/// \verbatim
/// The description continues.
/// \endverbatim
/// --
/// \param [out] I is a parameter.
/// \param J is a parameter.
/// \return void
void F(int I, int J);

/// Bonus comment on definition
void F(int I, int J) {}

// RUN: clang-doc --format=md --doxygen --public --extra-arg=-fmodules-ts -p %t %t/test.cpp -output=%t/docs


// RUN: cat %t/docs/./GlobalNamespace.md | FileCheck %s --check-prefix CHECK-0
// CHECK-0: # Global Namespace
// CHECK-0: ## Functions
// CHECK-0: ### F
// CHECK-0: *void F(int I, int J)*
// CHECK-0: *Defined at line 28 of test*
// CHECK-0: **brief** Brief description.
// CHECK-0:  Extended description that continues onto the next line.
// CHECK-0: <ul "class=test">
// CHECK-0: <li>
// CHECK-0:  Testing.</ul>
// CHECK-0:  The description continues.
// CHECK-0:  --
// CHECK-0: **I** [out]
// CHECK-0: **J**
// CHECK-0: **return** void
// CHECK-0:  Bonus comment on definition
