Fuzzing for LLVM-libc
---------------------

Fuzzing tests are used to ensure quality and security of LLVM-libc
implementations. 

Each fuzzing test lives under the fuzzing directory in a subdirectory
corresponding with the src layout. 

Currently we use system libc for functions that have yet to be implemented,
however as they are implemented the fuzzers will be changed to use our 
implementation to increase coverage for testing. 

Fuzzers will be run on `oss-fuzz <https://github.com/google/oss-fuzz>`_ and the
check-libc target will ensure that they build correctly. 
