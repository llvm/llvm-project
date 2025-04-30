#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
EXE=norm2.$(EXESUFFIX)

build:  $(SRC)/norm2.F90
	-$(RM) norm2.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	-$(RM) $(OBJ)
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	@echo ------------------------------------ building test $@
	$(FC) $(FFLAGS) $(LDFLAGS) $(SRC)/norm2.F90 check.$(OBJX) -o norm2.$(EXESUFFIX)
	$(FC) $(FFLAGS) $(LDFLAGS) $(SRC)/norm2.F90 check.$(OBJX) -r8 -o norm2R8.$(EXESUFFIX)

run:
	@echo ------------------------------------ executing test norm2
	norm2.$(EXESUFFIX)
	norm2R8.$(EXESUFFIX)

verify: ;

norm2.run: run
