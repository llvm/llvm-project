! 
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
! 

!          THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT
!   WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT
!   NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR
!   FITNESS FOR A PARTICULAR PURPOSE.
!


! iso_fortran_env.f90
! 32/64 bit  linux and windows.  Add further targets as required.

        module ISO_FORTRAN_ENV

	public

	integer CHARACTER_STORAGE_SIZE
	parameter (CHARACTER_STORAGE_SIZE = 8)
	integer ERROR_UNIT
	parameter (ERROR_UNIT = 0)
	integer FILE_STORAGE_SIZE
	parameter (FILE_STORAGE_SIZE = 8)
	integer INPUT_UNIT
	parameter (INPUT_UNIT = 5)
	integer IOSTAT_END
	parameter (IOSTAT_END = -1)
	integer IOSTAT_EOR
	parameter (IOSTAT_EOR = -2)
	integer NUMERIC_STORAGE_SIZE
	parameter (NUMERIC_STORAGE_SIZE = 32)
	integer OUTPUT_UNIT
	parameter (OUTPUT_UNIT = 6)

        integer IOSTAT_INQUIRE_INTERNAL_UNIT
        parameter (IOSTAT_INQUIRE_INTERNAL_UNIT=99)
    
	integer INT8
	parameter (INT8 = 1)
	integer INT16
	parameter (INT16 = 2)
	integer INT32
	parameter (INT32 = 4)
	integer INT64
	parameter (INT64 = 8)
	integer LOGICAL8
	parameter (LOGICAL8 = 1)
	integer LOGICAL16
	parameter (LOGICAL16 = 2)
	integer LOGICAL32
	parameter (LOGICAL32 = 4)
	integer LOGICAL64
	parameter (LOGICAL64 = 8)
	integer REAL32
	parameter (REAL32 = 4)
	integer REAL64
	parameter (REAL64 = 8)
	integer REAL128
#ifdef TARGET_SUPPORTS_QUADFP
        parameter (REAL128 = 16)
#else
        parameter (REAL128 = -1)
#endif

        integer INTEGER_KINDS(4)
        parameter (INTEGER_KINDS = (/INT8, INT16, INT32, INT64/))
        integer LOGICAL_KINDS(4)
        parameter (LOGICAL_KINDS = (/LOGICAL8, LOGICAL16, LOGICAL32, LOGICAL64/))
        integer REAL_KINDS(3)
        parameter (REAL_KINDS = (/REAL32, REAL64, REAL128/))

        end module  ISO_FORTRAN_ENV

