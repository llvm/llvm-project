#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test db01  ########


db01: run
	

build:  $(SRC)/db01.f
	-$(RM) db01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/db01.f -o db01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) db01.$(OBJX) check.$(OBJX) $(LIBS) -o db01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test db01
	db01.$(EXESUFFIX)

verify: ;

db01.run: run

