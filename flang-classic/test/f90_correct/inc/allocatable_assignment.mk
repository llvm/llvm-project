#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

EXE=allocatable_assignment.$(EXESUFFIX)

build:  $(SRC)/allocatable_assignment.f90
	-$(RM) allocatable_assignment.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	-$(RM) $(OBJ)
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	@echo ------------------------------------ building test $@
	$(FC) $(FFLAGS) $(LDFLAGS) $(SRC)/allocatable_assignment.f90 check.$(OBJX) -o allocatable_assignment.$(EXESUFFIX)

run:
	@echo ------------------------------------ executing test allocatable_assignment
	allocatable_assignment.$(EXESUFFIX)

verify: ;

allocatable_assignment.run: run
