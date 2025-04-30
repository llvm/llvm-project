#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pp35  ########


pp35: run
	

build:  $(SRC)/pp35.f90
	-$(RM) pp35.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pp35.f90 -o pp35.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pp35.$(OBJX) check.$(OBJX) $(LIBS) -o pp35.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pp35
	pp35.$(EXESUFFIX)

verify: ;

pp35.run: run

