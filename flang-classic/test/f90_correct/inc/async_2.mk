#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test async_2 ########


async_2: run

build:  $(SRC)/async_2.f90
	-$(RM) async_2.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/async_2.f90 -o async_2.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) async_2.$(OBJX) check.$(OBJX) $(LIBS) -o async_2.$(EXESUFFIX)


run:
	$(RM) ./*.txt
	@echo ------------------------------------ executing test async_2
	$(CP) $(SRC)/async_files/*.txt .
	@chmod 777 ./*.txt
	async_2.$(EXESUFFIX)

verify: ;
