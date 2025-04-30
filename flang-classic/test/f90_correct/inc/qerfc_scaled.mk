#
# Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qerfc_scaled ########


qerfc_scaled: run


build:	$(SRC)/qerfc_scaled.f08
	-$(RM) qerfc_scaled.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qerfc_scaled.f08 -o qerfc_scaled.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qerfc_scaled.$(OBJX) check.$(OBJX) $(LIBS) -o qerfc_scaled.$(EXESUFFIX)

run:
	@echo ------------------------------------ executing test qerfc_scaled
	qerfc_scaled.$(EXESUFFIX)

verify: ;

