preq_omega_ps kernel
Edited 03/18/2015
Amogh Simha

*kernel and supporting files
	-the preq_omega_ps subroutine is located in the prim_si_mod.F90 file
	-subroutine call is in the compute_and_apply_rhs subroutine in the prim_advance_mod.F90 file

*compilation and execution
	-Just download the enclosing directory
	-Run make

*verification
	-The make command will trigger the verification of the kernel.
	-It is considered to have passed verification if the tolerance for normalized RMS is less than 9.999999824516700E-015
	-Input data is provided by preq_omega_ps.1.0 preq_omega_ps.10.0, and preq_omega_ps.20.0

*performance measurement
	-The elapsed time in seconds is printed to stdout for each input file specified 

