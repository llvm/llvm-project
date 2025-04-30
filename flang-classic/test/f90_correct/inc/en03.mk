#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test en03  ########


en03: run


build:  $(SRC)/en03.f90
	-$(RM) en03.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/en03.f90 -o en03.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) en03.$(OBJX) check.$(OBJX) $(LIBS) -o en03.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test en03
	en03.$(EXESUFFIX)

verify: ;

en03.run: run

