#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre07  ########


pre07: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre07.f90
	-$(RM) pre07.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre07.f90 -o pre07.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre07.$(OBJX) check.$(OBJX) $(LIBS) -o pre07.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre07
	pre07.$(EXESUFFIX)

verify: ;

pre07.run: run

