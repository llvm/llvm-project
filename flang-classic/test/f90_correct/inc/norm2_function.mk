#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

EXE=norm2_function.$(EXESUFFIX)

build:  $(SRC)/norm2_function.F90
	-$(RM) norm2_function.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	-$(RM) $(OBJ)
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	@echo ------------------------------------ building test $@
	$(FC) $(FFLAGS) $(LDFLAGS) $(SRC)/norm2_function.F90 check.$(OBJX) -o norm2_function.$(EXESUFFIX)
	$(FC) $(FFLAGS) $(LDFLAGS) $(SRC)/norm2.F90 check.$(OBJX) -r8 -o norm2R8_function.$(EXESUFFIX)

run:
	@echo ------------------------------------ executing test norm2_function
	norm2_function.$(EXESUFFIX)
	norm2R8_function.$(EXESUFFIX)

verify: ;

norm2_function.run: run
