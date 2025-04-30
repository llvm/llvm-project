#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test scth  ########


scth: run
	

build:  $(SRC)/scth.f08
	-$(RM) scth.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/scth.f08 -o scth.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) scth.$(OBJX) check.$(OBJX) $(LIBS) -o scth.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test scth
	scth.$(EXESUFFIX)

verify: ;

scth.run: run

