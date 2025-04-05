/*
Opt-out of libclc mul_hi implementation for clspv.
clspv has an internal implementation that does not required using a bigger data size.
That implementation is based on OpMulExtended which is SPIR-V specific, thus it cannot be written in OpenCL-C.
*/
