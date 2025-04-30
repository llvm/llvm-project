#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ff00  ########


ff00: run
	

build:  $(SRC)/ff00.f
	-$(RM) ff00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ff00.f -o ff00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ff00.$(OBJX) check.$(OBJX) $(LIBS) -o ff00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ff00
	ff00.$(EXESUFFIX)

verify: ;

ff00.run: run

