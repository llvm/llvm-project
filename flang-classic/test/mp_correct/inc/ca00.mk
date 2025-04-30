#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
ca00: ca00.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN2) ./a.$(EXESUFFIX)
ca00.$(OBJX): $(SRC)/ca00.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/ca00.f
	@$(RM) ./a.$(EXESUFFIX) $(LOG)
	-$(FC) $(LDFLAGS) ca00.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: ca00
run: ;
