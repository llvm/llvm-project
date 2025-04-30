#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pp33  ########


pp33: run
	

build:  $(SRC)/pp33.f90
	-$(RM) pp33.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pp33.f90 -o pp33.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pp33.$(OBJX) check.$(OBJX) $(LIBS) -o pp33.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pp33
	pp33.$(EXESUFFIX)

verify: ;

pp33.run: run

