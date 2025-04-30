#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
pd00: pd00.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN2) ./a.$(EXESUFFIX) $(LOG)
pd00.$(OBJX): $(SRC)/pd00.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/pd00.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) pd00.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: pd00
run: ;
