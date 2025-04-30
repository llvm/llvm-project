#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

EXE=move_alloc.$(EXESUFFIX)

build:  $(SRC)/move_alloc.f90
	-$(RM) move_alloc.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	-$(RM) $(OBJ)
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	@echo ------------------------------------ building test $@
	$(FC) $(FFLAGS) $(LDFLAGS) $(SRC)/move_alloc.f90 check.$(OBJX) -o move_alloc.$(EXESUFFIX)

run:
	@echo ------------------------------------ executing test move_alloc
	move_alloc.$(EXESUFFIX)

verify: ;

move_alloc.run: run
