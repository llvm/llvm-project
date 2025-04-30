#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre17  ########


pre17: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre17.f90
	-$(RM) pre17.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre17.f90 -o pre17.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre17.$(OBJX) check.$(OBJX) $(LIBS) -o pre17.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre17
	pre17.$(EXESUFFIX)

verify: ;

pre17.run: run

