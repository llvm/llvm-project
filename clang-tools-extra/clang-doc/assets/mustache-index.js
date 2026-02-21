document.addEventListener("DOMContentLoaded", function() {
  const resizer = document.getElementById('resizer');
  const sidebar = document.querySelector('.sidebar');

  let isResizing = false;
  resizer.addEventListener('mousedown', (e) => { isResizing = true; });

  document.addEventListener('mousemove', (e) => {
    if (!isResizing)
      return;
    const newWidth = e.clientX;
    if (newWidth > 100 && newWidth < window.innerWidth - 100) {
      sidebar.style.width = `${newWidth}px`;
    }
  });

  document.addEventListener('mouseup', () => { isResizing = false; });

  document.querySelectorAll('pre code').forEach((el) => {
    hljs.highlightElement(el);
    el.classList.remove("hljs");
  });

  function getCharSize() {
    const testChar = document.createElement('span');
    testChar.className = "code-clang-doc"
    testChar.style.visibility = 'hidden';
    testChar.innerText = 'a';
    document.body.appendChild(testChar);
    const charWidth = testChar.getBoundingClientRect().width;
    document.body.removeChild(testChar);
    return charWidth;
  }

  function revertToSingleLine(func) {
    const paramsContainer = func.querySelectorAll('.params-vertical')
    const params = func.querySelectorAll('.param')
    paramsContainer.forEach(params => {
      params.style.display = "inline";
      params.style.paddingLeft = "0px";
    });
    params.forEach(param => {
      param.style.display = "inline";
      param.style.paddingLeft = "0px";
    });
  }

  const functions = document.querySelectorAll('.code-clang-doc');
  const content = document.querySelector('.content')
  const charSize = getCharSize();
  functions.forEach(func => {
    if(func.textContent.trim().length * charSize < content.clientWidth - 20)
      revertToSingleLine(func)
  });

  document.querySelectorAll('.sidebar-item-container').forEach(item => {
    item.addEventListener('click', function() {
      const anchor = item.getElementsByTagName("a");
      window.location.hash = anchor[0].getAttribute('href');
    });
  });
})
