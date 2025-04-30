#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fd00  ########


fd00: run
	

build:  $(SRC)/fd00.f
	-$(RM) fd00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fd00.f -o fd00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fd00.$(OBJX) check.$(OBJX) $(LIBS) -o fd00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fd00
	fd00.$(EXESUFFIX)

verify: ;

fd00.run: run

