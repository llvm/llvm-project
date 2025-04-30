Remap_q_ppm Kernel
Edited 02/24/2015
Amogh Simha

*kernel and supporting files
	-the remap_q_ppm subroutine is located in the prim_advection_mod.F90 file
	-subroutine call is in the same file at line 150 under the remap1 subroutine

*compilation and execution
	-Just download the enclosing directory
	-Run make

*verification
	-The make command will trigger the verification of the kernel.
	-It is considered to have passed verification if the tolerance for normalized RMS is less than 9.999999824516700E-015
	-Input data is provided by remap_q_ppm.1.0, remap_q_ppm.10.0, and remap_q_ppm.20.0

*performance measurement
	-The elapsed time in seconds is printed to stdout for each input file specified 

