#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre26  ########


pre26: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre26.f90
	-$(RM) pre26.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre26.f90 -o pre26.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre26.$(OBJX) check.$(OBJX) $(LIBS) -o pre26.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre26
	pre26.$(EXESUFFIX)

verify: ;

pre26.run: run

