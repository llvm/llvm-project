# Pull Request Checklist

Please ensure that your pull request addresses the following points:

- [ ] Have any of the opcode or instruction bit formatting changed in a breaking way?
    - If yes, please bump the ABI version in `llvm/lib/Target/Parasol/MCTargetDesc/ParasolELFObjectWriter.cpp`.
- [ ] Run `./format-parasol.sh` from the root directory to ensure your code is properly formatted.
- [ ] Updated the license-sunscreen-changes.txt with any changes to comply with the AGPLv3 and Apache licenses.
- [ ] Ensured that any files in this PR contain the Sunscreen license notice.

---

Thank you for your contribution to the Sunscreen LLVM fork!
