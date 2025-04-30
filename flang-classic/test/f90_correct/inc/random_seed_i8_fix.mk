#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

EXE=random_seed_i8_fix.$(EXESUFFIX)

build:  $(SRC)/random_seed_i8_fix.f90
	-$(RM) random_seed_i8_fix.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	-$(RM) $(OBJ)
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	@echo ------------------------------------ building test $@
	$(FC) $(FFLAGS) -i8 $(LDFLAGS) $(SRC)/random_seed_i8_fix.f90 check.$(OBJX) -o random_seed_i8_fix.$(EXESUFFIX)

run:
	@echo ------------------------------------ executing test random_seed_i8_fix
	random_seed_i8_fix.$(EXESUFFIX)

verify: ;

random_seed_i8_fix.run: run
