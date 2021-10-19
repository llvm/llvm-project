// RUN: %check_clang_tidy %s altera-kernel-name-restriction %t -- -- -I%S/Inputs/altera-kernel-name-restriction
// RUN: %check_clang_tidy -check-suffix=UPPERCASE %s altera-kernel-name-restriction %t -- -- -I%S/Inputs/altera-kernel-name-restriction/uppercase -DUPPERCASE

#ifdef UPPERCASE
// The warning should be triggered regardless of capitalization
#include "KERNEL.cl"
// CHECK-MESSAGES-UPPERCASE: :[[@LINE-1]]:1: warning: including 'KERNEL.cl' may cause additional compilation errors due to the name of the kernel source file; consider renaming the included kernel source file [altera-kernel-name-restriction]
#include "vERILOG.cl"
// CHECK-MESSAGES-UPPERCASE: :[[@LINE-1]]:1: warning: including 'vERILOG.cl' may cause additional compilation errors due to the name of the kernel source file; consider renaming the included kernel source file [altera-kernel-name-restriction]
#include "VHDL.cl"
// CHECK-MESSAGES-UPPERCASE: :[[@LINE-1]]:1: warning: including 'VHDL.cl' may cause additional compilation errors due to the name of the kernel source file; consider renaming the included kernel source file [altera-kernel-name-restriction]
#else 
// These are the banned kernel filenames, and should trigger warnings
#include "kernel.cl"
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: including 'kernel.cl' may cause additional compilation errors due to the name of the kernel source file; consider renaming the included kernel source file [altera-kernel-name-restriction]
#include "Verilog.cl"
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: including 'Verilog.cl' may cause additional compilation errors due to the name of the kernel source file; consider renaming the included kernel source file [altera-kernel-name-restriction]
#include "vhdl.CL"
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: including 'vhdl.CL' may cause additional compilation errors due to the name of the kernel source file; consider renaming the included kernel source file [altera-kernel-name-restriction]


// The warning should be triggered if the names are within a directory
#include "some/dir/kernel.cl"
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: including 'kernel.cl' may cause additional compilation errors due to the name of the kernel source file; consider renaming the included kernel source file [altera-kernel-name-restriction]
#include "somedir/verilog.cl"
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: including 'verilog.cl' may cause additional compilation errors due to the name of the kernel source file; consider renaming the included kernel source file [altera-kernel-name-restriction]
#include "otherdir/vhdl.cl"
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: including 'vhdl.cl' may cause additional compilation errors due to the name of the kernel source file; consider renaming the included kernel source file [altera-kernel-name-restriction]

// There are no FIX-ITs for the altera-kernel-name-restriction lint check

// The following include directives shouldn't trigger the warning
#include "otherthing.cl"
#include "thing.h"

// It doesn't make sense to have kernel.h, verilog.h, or vhdl.h as filenames
// without the corresponding .cl files, but the Altera Programming Guide doesn't
// explicitly forbid it.
#include "kernel.h"
#include "verilog.h"
#include "vhdl.h"

// The files can still have the forbidden names in them, so long as they're not
// the entire file name, and are not the kernel source file name.
#include "some_kernel.cl"
#include "other_Verilog.cl"
#include "vhdl_number_two.cl"

// Naming a directory kernel.cl, verilog.cl, or vhdl.cl is not explicitly
// forbidden in the Altera Programming Guide either.
#include "some/kernel.cl/foo.h"
#include "some/verilog.cl/foo.h"
#include "some/vhdl.cl/foo.h"
#endif

