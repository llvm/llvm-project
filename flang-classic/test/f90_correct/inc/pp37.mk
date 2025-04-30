#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pp37  ########


pp37: run
	

build:  $(SRC)/pp37.f90
	-$(RM) pp37.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pp37.f90 -o pp37.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pp37.$(OBJX) check.$(OBJX) $(LIBS) -o pp37.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pp37
	pp37.$(EXESUFFIX)

verify: ;

pp37.run: run

