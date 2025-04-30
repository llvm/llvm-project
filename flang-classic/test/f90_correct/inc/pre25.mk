#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre25  ########


pre25: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre25.f90
	-$(RM) pre25.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre25.f90 -o pre25.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre25.$(OBJX) check.$(OBJX) $(LIBS) -o pre25.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre25
	pre25.$(EXESUFFIX)

verify: ;

pre25.run: run

