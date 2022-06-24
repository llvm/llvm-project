
//// Verify that valid options for the -mrecip flag are passed through and invalid options cause an error.


//// If there are no options, convert to 'all'.

// RUN: %clang -### -S %s -mrecip  2>&1 | FileCheck --check-prefix=RECIP0 %s
// RECIP0: "-mrecip=all"

//// Check options that cover all types.

// RUN: %clang -### -S %s -mrecip=all  2>&1 | FileCheck --check-prefix=RECIP1 %s
// RECIP1: "-mrecip=all"

// RUN: %clang -### -S %s -mrecip=default  2>&1 | FileCheck --check-prefix=RECIP2 %s
// RECIP2: "-mrecip=default"

// RUN: %clang -### -S %s -mrecip=none  2>&1 | FileCheck --check-prefix=RECIP3 %s
// RECIP3: "-mrecip=none"

//// Check options that do not specify float or double.

// RUN: %clang -### -S %s -mrecip=vec-sqrt  2>&1 | FileCheck --check-prefix=RECIP4 %s
// RECIP4: "-mrecip=vec-sqrt"

// RUN: %clang -### -S %s -mrecip=!div,vec-div  2>&1 | FileCheck --check-prefix=RECIP5 %s
// RECIP5: "-mrecip=!div,vec-div"

//// Check individual option types.

// RUN: %clang -### -S %s -mrecip=vec-sqrtd  2>&1 | FileCheck --check-prefix=RECIP6 %s
// RECIP6: "-mrecip=vec-sqrtd"

// RUN: %clang -### -S %s -mrecip=!divf  2>&1 | FileCheck --check-prefix=RECIP7 %s
// RECIP7: "-mrecip=!divf"

// RUN: %clang -### -S %s -mrecip=divf,sqrtd,vec-divd,vec-sqrtf  2>&1 | FileCheck --check-prefix=RECIP8 %s
// RECIP8: "-mrecip=divf,sqrtd,vec-divd,vec-sqrtf"

// RUN: %clang -### -S %s -mrecip=vec-sqrth  2>&1 | FileCheck --check-prefix=RECIP18 %s
// RECIP18: "-mrecip=vec-sqrth"

//// Check optional refinement step specifiers.

// RUN: %clang -### -S %s -mrecip=all:1  2>&1 | FileCheck --check-prefix=RECIP9 %s
// RECIP9: "-mrecip=all:1"

// RUN: %clang -### -S %s -mrecip=sqrtf:3  2>&1 | FileCheck --check-prefix=RECIP10 %s
// RECIP10: "-mrecip=sqrtf:3"

// RUN: %clang -### -S %s -mrecip=div:5  2>&1 | FileCheck --check-prefix=RECIP11 %s
// RECIP11: "-mrecip=div:5"

// RUN: %clang -### -S %s -mrecip=divd:1,!sqrtf:2,vec-divf:9,vec-sqrtd:0  2>&1 | FileCheck --check-prefix=RECIP12 %s
// RECIP12: "-mrecip=divd:1,!sqrtf:2,vec-divf:9,vec-sqrtd:0"

// RUN: %clang -### -S %s -mrecip=sqrth:2  2>&1 | FileCheck --check-prefix=RECIP19 %s
// RECIP19: "-mrecip=sqrth:2"

//// Check invalid parameters.

// RUN: %clang -### -S %s -mrecip=bogus  2>&1 | FileCheck --check-prefix=RECIP13 %s
// RECIP13: error: unknown argument

// RUN: %clang -### -S %s -mrecip=divd:1,divd  2>&1 | FileCheck --check-prefix=RECIP14 %s
// RECIP14: error: invalid value 

// RUN: %clang -### -S %s -mrecip=sqrt,sqrtf  2>&1 | FileCheck --check-prefix=RECIP15 %s
// RECIP15: error: invalid value 

// RUN: %clang -### -S %s -mrecip=+default:10  2>&1 | FileCheck --check-prefix=RECIP16 %s
// RECIP16: error: invalid value 

// RUN: %clang -### -S %s -mrecip=!vec-divd:  2>&1 | FileCheck --check-prefix=RECIP17 %s
// RECIP17: error: invalid value 

// RUN: %clang -### -S %s -mrecip=divh:1,divh  2>&1 | FileCheck --check-prefix=RECIP20 %s
// RECIP20: error: invalid value 

// RUN: %clang -### -S %s -mrecip=divh,div  2>&1 | FileCheck --check-prefix=RECIP21 %s
// RECIP21: error: invalid value 

// RUN: %clang -### -S %s -mrecip=sqrt,sqrth  2>&1 | FileCheck --check-prefix=RECIP22 %s
// RECIP22: error: invalid value 
