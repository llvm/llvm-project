#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

EXE=norm2_single_precision.$(EXESUFFIX)

build:  $(SRC)/norm2_single_precision.F90
	-$(RM) norm2_single_precision.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	-$(RM) $(OBJ)
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	@echo ------------------------------------ building test $@
	$(FC) $(FFLAGS) $(LDFLAGS) $(SRC)/norm2_single_precision.F90 check.$(OBJX) -o norm2_single_precision.$(EXESUFFIX)

run:
	@echo ------------------------------------ executing test Norm2 Single Precicion
	norm2_single_precision.$(EXESUFFIX)

verify: ;

norm2_single_precision.run: run
