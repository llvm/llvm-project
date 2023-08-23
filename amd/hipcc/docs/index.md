# HIPCC

`hipcc` is a compiler driver utility that will call clang or nvcc, depending on target, and pass the appropriate include and library options for the target compiler and HIP infrastructure. 

There is both a Perl version, and a C++ executable version of the hipcc/hipconfig compiler driver utilities provided. Currently, by default the Perl version is used when 'hipcc' is called. To enable the C++ executable versions, set the environment variable `HIP_USE_PERL_SCRIPTS=0`.
