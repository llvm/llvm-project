#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre02  ########


pre02: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre02.f90
	-$(RM) pre02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre02.f90 -o pre02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre02.$(OBJX) check.$(OBJX) $(LIBS) -o pre02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre02
	pre02.$(EXESUFFIX)

verify: ;

pre02.run: run

