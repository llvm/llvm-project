#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
build:  $(SRC)/minmaxloc_back.f90
	-$(RM) minmaxloc_back.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	-$(RM) $(OBJ)
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	@echo ------------------------------------ building test $@
	$(FC) $(FFLAGS) $(LDFLAGS) $(SRC)/minmaxloc_back.f90 check.$(OBJX) -o minmaxloc_back.$(EXESUFFIX)

run:
	@echo ------------------------------------ executing test minmaxloc_back
	minmaxloc_back.$(EXESUFFIX)

verify: ;
