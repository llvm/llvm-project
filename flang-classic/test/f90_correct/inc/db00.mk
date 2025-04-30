#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test db00  ########


db00: run
	

build:  $(SRC)/db00.f
	-$(RM) db00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/db00.f -o db00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) db00.$(OBJX) check.$(OBJX) $(LIBS) -o db00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test db00
	db00.$(EXESUFFIX)

verify: ;

db00.run: run

