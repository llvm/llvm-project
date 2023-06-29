/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#ifndef OCLC_H
#define OCLC_H

// These constants are used to control behavior of the libraries which
// check them.
//
// The current list of controls is as follows:
//
//    __constant bool __oclc_finite_only_opt
//        - the application will only pass finite arguments and expects only finite results
//
//    __constant bool __oclc_unsafe_math_opt
//        - the aopplication accepts optimizations that may lower the accuracy of the results
//
//    __constant bool __oclc_daz_opt(void)
//        - the application allows subnormal inputs or outputs to be flushed to zero
//
//    __constant bool __oclc_correctly_rounded_sqrt32(void)
//        - the application is expecting sqrt(float) to produce a correctly rounded result
//
//    __constant bool __oclc_wavefrontsize64
//        - the application is being compiled for a wavefront size of 64
//
//    __constant int __oclc_ISA_version
//        - the ISA version of the target device
//
//    __constant int __oclc_ABI_version
//        - the ABI version the application is being compiled for
//
// it is expected that the implementation provides these as if declared from the following
// C code:
//
//     const bool int __oclc_... = 0; // Or 1
//
// allowing them and any control flow associated with them to be optimized away

extern const __constant bool __oclc_finite_only_opt;
extern const __constant bool __oclc_unsafe_math_opt;
extern const __constant bool __oclc_daz_opt;
extern const __constant bool __oclc_correctly_rounded_sqrt32;
extern const __constant bool __oclc_wavefrontsize64;
extern const __constant uint __oclc_wavefrontsize_log2;
extern const __constant int __oclc_ISA_version;
extern const __constant int __oclc_ABI_version;

#define OCLC_WAVEFRONT_SIZE (1u << __oclc_wavefrontsize_log2)


#endif // OCLC_H
