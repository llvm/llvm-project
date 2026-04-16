// RUN: rm -f %t.h %t.pch
// RUN: echo "int variable_del_cern = 42;" > %t.h
// RUN: %clang_cc1 -x c++-header -emit-pch -o %t.pch %t.h
// RUN: rm %t.h
// RUN: %clang_cc1 -fno-validate-pch -ast-dump -include-pch %t.pch %s

// This test ensures that Clang does not access the filesystem when
// handling SourceLocations originating from a PCH.
//
// After generating the PCH, the original header is removed. If Clang
// attempts to access it (e.g. via MeasureTokenLength), the test will fail.

int function() { 
    return variable_del_cern; 
}
