#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre03  ########


pre03: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre03.f90
	-$(RM) pre03.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre03.f90 -o pre03.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre03.$(OBJX) check.$(OBJX) $(LIBS) -o pre03.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre03
	pre03.$(EXESUFFIX)

verify: ;

pre03.run: run

