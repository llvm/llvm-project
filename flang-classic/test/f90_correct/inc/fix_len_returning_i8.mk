#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

EXE1=fix_len_returning_i8.$(EXESUFFIX)
EXE2=fix_len_returning_i8_di8.$(EXESUFFIX)

build:  $(SRC)/fix_len_returning_i8.f90
	-$(RM) fix_len_returning_i8.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	-$(RM) $(OBJ)
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	@echo ------------------------------------ building test $@
	$(FC) $(FFLAGS) $(LDFLAGS) $(SRC)/fix_len_returning_i8.f90 check.$(OBJX) -o $(EXE1)
	$(FC) $(FFLAGS) -fdefault-integer-8 $(LDFLAGS) $(SRC)/fix_len_returning_i8.f90 check.$(OBJX) -o $(EXE2)

run:
	@echo ------------------------------------ executing test $(EXE1)
	$(EXE1)
	@echo ------------------------------------ executing test $(EXE2)
	$(EXE2)

verify: ;

fix_len_returning_i8.run: run
