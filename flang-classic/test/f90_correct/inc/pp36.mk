#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pp36  ########


pp36: run
	

build:  $(SRC)/pp36.f90
	-$(RM) pp36.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pp36.f90 -o pp36.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pp36.$(OBJX) check.$(OBJX) $(LIBS) -o pp36.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pp36
	pp36.$(EXESUFFIX)

verify: ;

pp36.run: run

