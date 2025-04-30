#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pp34  ########


pp34: run
	

build:  $(SRC)/pp34.f90
	-$(RM) pp34.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pp34.f90 -o pp34.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pp34.$(OBJX) check.$(OBJX) $(LIBS) -o pp34.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pp34
	pp34.$(EXESUFFIX)

verify: ;

pp34.run: run

