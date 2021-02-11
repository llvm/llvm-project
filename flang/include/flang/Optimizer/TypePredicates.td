//===-- TypePredicates.td ----------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "flang/Optimizer/Dialect/FIRTypes.td"

#ifndef FORTRAN_FIR_OPTIMIZER_TYPEPREDICATES
#define FORTRAN_FIR_OPTIMIZER_TYPEPREDICATES

def fir_IntegerType : Type<CPred<"$_self.isa<fir::IntegerType>()">,
    "FIR integer type">;

def AnyIntegerLike : TypeConstraint<Or<[SignlessIntegerLike.predicate,
    AnySignedInteger.predicate, fir_IntegerType.predicate]>, "any integer">;

def AnyIntegerType : Type<AnyIntegerLike.predicate, "any integer">;

// Reference to an entity type
def fir_ReferenceType : Type<CPred<"$_self.isa<fir::ReferenceType>()">,
    "reference type">;

// Reference to an ALLOCATABLE attribute type
def fir_HeapType : Type<CPred<"$_self.isa<fir::HeapType>()">,
    "allocatable type">;

// Reference to a POINTER attribute type
def fir_PointerType : Type<CPred<"$_self.isa<fir::PointerType>()">,
    "pointer type">;

// Reference types
def AnyReferenceLike : TypeConstraint<Or<[fir_ReferenceType.predicate,
    fir_HeapType.predicate, fir_PointerType.predicate]>, "any reference">;

// A LEN parameter (in a RecordType) argument's type
def fir_LenType : Type<CPred<"$_self.isa<fir::LenType>()">,
    "LEN parameter type">;

def AnyCoordinateLike : TypeConstraint<Or<[AnySignlessInteger.predicate,
    Index.predicate, fir_IntegerType.predicate, fir_FieldType.predicate,
    fir_LenType.predicate]>, "any coordinate index">;

def AnyCoordinateType : Type<AnyCoordinateLike.predicate, "coordinate type">;

// Reference and Box types
def AnyRefOrBox : TypeConstraint<Or<[fir_ReferenceType.predicate,
    fir_HeapType.predicate, fir_PointerType.predicate, fir_BoxType.predicate]>,
    "any reference or box">;

#endif
