#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre18  ########


pre18: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre18.f90
	-$(RM) pre18.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre18.f90 -o pre18.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre18.$(OBJX) check.$(OBJX) $(LIBS) -o pre18.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre18
	pre18.$(EXESUFFIX)

verify: ;

pre18.run: run

