#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre20  ########


pre20: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre20.f90
	-$(RM) pre20.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre20.f90 -o pre20.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre20.$(OBJX) check.$(OBJX) $(LIBS) -o pre20.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre20
	pre20.$(EXESUFFIX)

verify: ;

pre20.run: run

