#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre04  ########


pre04: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre04.f90
	-$(RM) pre04.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre04.f90 -o pre04.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre04.$(OBJX) check.$(OBJX) $(LIBS) -o pre04.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre04
	pre04.$(EXESUFFIX)

verify: ;

pre04.run: run

