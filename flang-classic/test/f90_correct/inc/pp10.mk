#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pp10  ########


pp10: run
	

build:  $(SRC)/pp10.f90
	-$(RM) pp10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pp10.f90 -o pp10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pp10.$(OBJX) check.$(OBJX) $(LIBS) -o pp10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pp10
	pp10.$(EXESUFFIX)

verify: ;

pp10.run: run

