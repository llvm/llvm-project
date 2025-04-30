Limiter_optim_iter_full Kernel
Edited 03/03/2015
Amogh Simha

*kernel and supporting files
	-the limiter_optim_iter_full subroutine is located in the prim_advection_mod.F90 file
	-subroutine call is in the same file in the euler_step subroutine

*compilation and execution
	-Just download the enclosing directory
	-Run make

*verification
	-The make command will trigger the verification of the kernel.
	-It is considered to have passed verification if the tolerance for normalized RMS is less than 9.999999824516700E-015
	-Input data is provided by limiter_optim_iter_full.1.0, limiter_optim_iter_full.10.0, and limiter_optim_iter_full.20.0

*performance measurement
	-The elapsed time in seconds is printed to stdout for each input file specified 

