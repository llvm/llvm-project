#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre31  ########


pre31: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre31.f90
	-$(RM) pre31.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre31.f90 -o pre31.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre31.$(OBJX) check.$(OBJX) $(LIBS) -o pre31.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre31
	pre31.$(EXESUFFIX)

verify: ;

pre31.run: run

