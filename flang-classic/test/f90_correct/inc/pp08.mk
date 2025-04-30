#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pp08  ########


pp08: run
	

build:  $(SRC)/pp08.f90
	-$(RM) pp08.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pp08.f90 -o pp08.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pp08.$(OBJX) check.$(OBJX) $(LIBS) -o pp08.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pp08
	pp08.$(EXESUFFIX)

verify: ;

pp08.run: run

