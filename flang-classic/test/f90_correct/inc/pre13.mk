#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre13  ########


pre13: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre13.f90
	-$(RM) pre13.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre13.f90 -o pre13.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre13.$(OBJX) check.$(OBJX) $(LIBS) -o pre13.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre13
	pre13.$(EXESUFFIX)

verify: ;

pre13.run: run

