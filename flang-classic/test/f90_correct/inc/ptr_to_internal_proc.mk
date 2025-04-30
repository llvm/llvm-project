#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

EXE=ptr_to_internal_proc.$(EXESUFFIX)

build:  $(SRC)/ptr_to_internal_proc.f90
	-$(RM) ptr_to_internal_proc.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	-$(RM) $(OBJ)
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	@echo ------------------------------------ building test $@
	$(FC) $(FFLAGS) $(LDFLAGS) $(SRC)/ptr_to_internal_proc.f90 check.$(OBJX) -o ptr_to_internal_proc.$(EXESUFFIX)

run:
	@echo ------------------------------------ executing test ptr_to_internal_proc
	ptr_to_internal_proc.$(EXESUFFIX)

verify: ;

ptr_to_internal_proc.run: run
