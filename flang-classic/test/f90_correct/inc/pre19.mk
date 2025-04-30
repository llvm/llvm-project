#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre19  ########


pre19: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre19.f90
	-$(RM) pre19.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre19.f90 -o pre19.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre19.$(OBJX) check.$(OBJX) $(LIBS) -o pre19.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre19
	pre19.$(EXESUFFIX)

verify: ;

pre19.run: run

