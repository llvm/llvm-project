#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pp15  ########


pp15: run
	

build:  $(SRC)/pp15.f90
	-$(RM) pp15.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pp15.f90 -o pp15.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pp15.$(OBJX) check.$(OBJX) $(LIBS) -o pp15.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pp15
	pp15.$(EXESUFFIX)

verify: ;

pp15.run: run

