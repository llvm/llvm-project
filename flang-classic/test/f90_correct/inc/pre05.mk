#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre05  ########


pre05: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre05.f90
	-$(RM) pre05.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre05.f90 -o pre05.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre05.$(OBJX) check.$(OBJX) $(LIBS) -o pre05.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre05
	pre05.$(EXESUFFIX)

verify: ;

pre05.run: run

