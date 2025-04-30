#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

EXE=transpose_init.$(EXESUFFIX)

build:  $(SRC)/transpose_init.f90
	-$(RM) transpose_init.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	-$(RM) $(OBJ)
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	@echo ------------------------------------ building test $@
	$(FC) $(FFLAGS) $(LDFLAGS) $(SRC)/transpose_init.f90 check.$(OBJX) -o transpose_init.$(EXESUFFIX)

run:
	@echo ------------------------------------ executing test transpose_init
	transpose_init.$(EXESUFFIX)

verify: ;

transpose_init.run: run
